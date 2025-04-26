package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_1_0_Test {

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testError() {
        AnyWrapperMsgGenerator wrapper = Mockito.mock(AnyWrapperMsgGenerator.class);
        String result = wrapper.error("Hello, World!");
        assertEquals("Hello, World!", result);
    }
}
