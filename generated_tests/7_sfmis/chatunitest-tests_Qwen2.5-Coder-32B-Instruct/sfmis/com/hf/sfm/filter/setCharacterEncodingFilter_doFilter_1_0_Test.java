package com.hf.sfm.filter;

import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.Filter;

public class setCharacterEncodingFilter_doFilter_1_0_Test {

    @Test
    public void testDoFilter() throws IOException, ServletException {
        // Arrange
        ServletRequest mockRequest = mock(ServletRequest.class);
        ServletResponse mockResponse = mock(ServletResponse.class);
        FilterChain mockChain = mock(FilterChain.class);
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        // Act
        filter.doFilter(mockRequest, mockResponse, mockChain);
        // Assert
        verify(mockRequest, times(1)).setCharacterEncoding("UTF-8");
        verify(mockChain, times(1)).doFilter(mockRequest, mockResponse);
    }
}
