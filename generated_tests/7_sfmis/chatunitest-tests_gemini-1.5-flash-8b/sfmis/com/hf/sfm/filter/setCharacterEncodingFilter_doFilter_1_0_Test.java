package com.hf.sfm.filter;

import javax.servlet.FilterChain;
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
import javax.servlet.FilterConfig;

public class setCharacterEncodingFilter_doFilter_1_0_Test {

    @Test
    public void testDoFilter_encodingSet() throws IOException, ServletException {
        // Mock objects
        ServletRequest request = Mockito.mock(ServletRequest.class);
        ServletResponse response = Mockito.mock(ServletResponse.class);
        FilterChain chain = Mockito.mock(FilterChain.class);
        // Create instance of the class under test
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        // Invoke the method under test
        filter.doFilter(request, response, chain);
        // Verify that the character encoding was set
        ArgumentCaptor<String> encodingCaptor = ArgumentCaptor.forClass(String.class);
        Mockito.verify(request).setCharacterEncoding(encodingCaptor.capture());
        assertEquals("UTF-8", encodingCaptor.getValue());
        // Verify that the chain was called
        Mockito.verify(chain).doFilter(request, response);
    }
}
