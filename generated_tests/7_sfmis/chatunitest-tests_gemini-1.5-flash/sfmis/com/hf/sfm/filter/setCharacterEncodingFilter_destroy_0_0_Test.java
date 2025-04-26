package com.hf.sfm.filter;

import javax.servlet.*;
import java.io.IOException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class setCharacterEncodingFilter_destroy_0_0_Test {

    @Mock
    private FilterConfig filterConfig;

    @Test
    void testDestroy() {
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        try {
            filter.init(filterConfig);
        } catch (ServletException e) {
            fail("Unexpected ServletException during init: " + e.getMessage());
        }
        assertDoesNotThrow(filter::destroy);
    }

    @Test
    void testDoFilter_EncodingSet() throws ServletException, IOException {
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        filter.init(filterConfig);
        ServletRequest request = mock(ServletRequest.class);
        ServletResponse response = mock(ServletResponse.class);
        FilterChain chain = mock(FilterChain.class);
        when(filterConfig.getInitParameter("encoding")).thenReturn("UTF-8");
        filter.doFilter(request, response, chain);
        verify(request).setCharacterEncoding("UTF-8");
        verify(chain).doFilter(request, response);
    }

    @Test
    void testDoFilter_EncodingNotSet() throws ServletException, IOException {
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        filter.init(filterConfig);
        ServletRequest request = mock(ServletRequest.class);
        ServletResponse response = mock(ServletResponse.class);
        FilterChain chain = mock(FilterChain.class);
        when(filterConfig.getInitParameter("encoding")).thenReturn(null);
        filter.doFilter(request, response, chain);
        verify(request, never()).setCharacterEncoding(anyString());
        verify(chain).doFilter(request, response);
    }

    @Test
    void testDoFilter_Exception() throws ServletException, IOException {
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        filter.init(filterConfig);
        ServletRequest request = mock(ServletRequest.class);
        ServletResponse response = mock(ServletResponse.class);
        FilterChain chain = mock(FilterChain.class);
        when(filterConfig.getInitParameter("encoding")).thenReturn("UTF-8");
        doThrow(new ServletException("Simulated Exception")).when(chain).doFilter(request, response);
        assertThrows(ServletException.class, () -> filter.doFilter(request, response, chain));
    }

    static class setCharacterEncodingFilter implements Filter {

        private FilterConfig filterConfig;

        @Override
        public void init(FilterConfig filterConfig) throws ServletException {
            this.filterConfig = filterConfig;
        }

        @Override
        public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
            String encoding = filterConfig.getInitParameter("encoding");
            if (encoding != null) {
                request.setCharacterEncoding(encoding);
            }
            chain.doFilter(request, response);
        }

        @Override
        public void destroy() {
            // No-op for this test
        }
    }
}
