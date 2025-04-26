package com.hf.sfm.filter;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.IOException;
import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;

class setCharacterEncodingFilter_destroy_0_1_Test {

    @Test
    void testDestroy() {
        setCharacterEncodingFilter filter = Mockito.mock(setCharacterEncodingFilter.class);
        Mockito.doNothing().when(filter).destroy();
        filter.destroy();
    }
}
