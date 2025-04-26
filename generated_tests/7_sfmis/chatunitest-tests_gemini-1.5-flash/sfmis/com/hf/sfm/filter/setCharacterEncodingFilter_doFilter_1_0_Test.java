package com.hf.sfm.filter;

import javax.servlet.*;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class setCharacterEncodingFilter_doFilter_1_0_Test {

    @Test
    void testDoFilter() throws ServletException, IOException, NoSuchFieldException, IllegalAccessException {
        // Create mock objects
        ServletRequest request = mock(HttpServletRequest.class);
        ServletResponse response = mock(HttpServletResponse.class);
        FilterChain chain = mock(FilterChain.class);
        // Create instance of the class under test
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        // Invoke the method under test
        filter.doFilter(request, response, chain);
        // Verify that the character encoding was set
        verify(request).setCharacterEncoding("UTF-8");
        // Verify that the filter chain was invoked
        verify(chain).doFilter(request, response);
    }

    @Test
    void testDoFilter_ExceptionHandling() throws ServletException, IOException {
        // Create mock objects
        ServletRequest request = mock(HttpServletRequest.class);
        ServletResponse response = mock(HttpServletResponse.class);
        FilterChain chain = mock(FilterChain.class);
        doThrow(new IOException("Simulated IO Exception")).when(chain).doFilter(request, response);
        // Create instance of the class under test
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        // Invoke the method under test and assert exception is thrown
        assertThrows(IOException.class, () -> filter.doFilter(request, response, chain));
        // Verify that the character encoding was still attempted to be set even with exception
        verify(request).setCharacterEncoding("UTF-8");
    }
}
