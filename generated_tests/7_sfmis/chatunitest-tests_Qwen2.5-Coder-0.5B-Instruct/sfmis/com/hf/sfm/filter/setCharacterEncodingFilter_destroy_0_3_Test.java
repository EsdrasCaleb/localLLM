package com.hf.sfm.filter;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.IOException;
import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class setCharacterEncodingFilter_destroy_0_3_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestSetCharacterEncodingFilterDestroy_0_3 {

        @Mock
        private Filter filter;

        @Test
        public void testDestroy() throws ServletException, IOException {
            // Act
            filter.destroy();
            // Assert
            verify(filter).destroy();
        }
    }
}
