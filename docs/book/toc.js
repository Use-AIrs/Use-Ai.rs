// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="index.html">Introduction</a></li><li class="chapter-item expanded "><a href="tooling/index.html"><strong aria-hidden="true">1.</strong> Tooling</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="tooling/testai.html"><strong aria-hidden="true">1.1.</strong> Test AI</a></li><li class="chapter-item expanded "><a href="tooling/useai.html"><strong aria-hidden="true">1.2.</strong> Use AI</a></li></ol></li><li class="chapter-item expanded "><a href="store/store.html"><strong aria-hidden="true">2.</strong> Store</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="store/config.html"><strong aria-hidden="true">2.1.</strong> Config</a></li><li class="chapter-item expanded "><a href="store/db.html"><strong aria-hidden="true">2.2.</strong> MangoDB</a></li></ol></li><li class="chapter-item expanded "><a href="stage/stage.html"><strong aria-hidden="true">3.</strong> Stage</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="stage/config.html"><strong aria-hidden="true">3.1.</strong> Config</a></li><li class="chapter-item expanded "><a href="stage/input.html"><strong aria-hidden="true">3.2.</strong> Data Input</a></li><li class="chapter-item expanded "><a href="stage/transfer.html"><strong aria-hidden="true">3.3.</strong> Transfer</a></li><li class="chapter-item expanded "><a href="stage/output.html"><strong aria-hidden="true">3.4.</strong> Data Output</a></li></ol></li><li class="chapter-item expanded "><a href="calc/calc.html"><strong aria-hidden="true">4.</strong> Calculator</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="calc/config.html"><strong aria-hidden="true">4.1.</strong> Config</a></li><li class="chapter-item expanded "><a href="calc/model.html"><strong aria-hidden="true">4.2.</strong> Model</a></li><li class="chapter-item expanded "><a href="calc/operation.html"><strong aria-hidden="true">4.3.</strong> Operation</a></li></ol></li><li class="chapter-item expanded "><a href="macros/macros.html"><strong aria-hidden="true">5.</strong> Proc Macros</a></li><li class="chapter-item expanded "><a href="blog/index.html"><strong aria-hidden="true">6.</strong> Blog</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="blog/5.html"><strong aria-hidden="true">6.1.</strong> Introducing Action Space</a></li><li class="chapter-item expanded "><a href="blog/4.html"><strong aria-hidden="true">6.2.</strong> Short Update</a></li><li class="chapter-item expanded "><a href="blog/3.html"><strong aria-hidden="true">6.3.</strong> New Blog</a></li><li class="chapter-item expanded "><a href="blog/2.html"><strong aria-hidden="true">6.4.</strong> Approach</a></li><li class="chapter-item expanded "><a href="blog/1.html"><strong aria-hidden="true">6.5.</strong> Tool Demo</a></li><li class="chapter-item expanded "><a href="blog/0.html"><strong aria-hidden="true">6.6.</strong> Welcome</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
