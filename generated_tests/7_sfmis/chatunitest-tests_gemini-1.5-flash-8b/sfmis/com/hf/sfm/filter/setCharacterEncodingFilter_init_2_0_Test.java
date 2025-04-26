package com.hf.sfm.filter;

import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.IOException;
import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;

public class setCharacterEncodingFilter_init_2_0_Test {

    @Test
    public void testInit_NoException() {
        FilterConfig filterConfigMock = Mockito.mock(FilterConfig.class);
        setCharacterEncodingFilter filter = new setCharacterEncodingFilter();
        try {
            filter.init(filterConfigMock);
        } catch (ServletException e) {
            // Should not throw an exception
            throw new AssertionError("Unexpected ServletException", e);
        }
    }
}
