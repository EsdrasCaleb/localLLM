package com.hf.sfm.filter;

import java.io.IOException;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.Filter;
import javax.servlet.FilterConfig;

class setCharacterEncodingFilter_doFilter_1_2_Test {

    @Test
    void testDoFilter() throws IOException, ServletException {
        // Arrange
        ServletRequest request = mock(ServletRequest.class);
        ServletResponse response = mock(ServletResponse.class);
        FilterChain chain = mock(FilterChain.class);
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        // Act
        filter.doFilter(request, response, chain);
        // Assert
        verify(request).setCharacterEncoding("UTF-8");
        InOrder inOrder = inOrder(request, chain);
        inOrder.verify(request).setCharacterEncoding("UTF-8");
        inOrder.verify(chain).doFilter(request, response);
    }
}
