package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_1_0_Test {

    // Test class
    @Test
    public void testError() {
        AnyWrapperMsgGenerator wrapper = new AnyWrapperMsgGenerator();
        String res = wrapper.error("Hello");
        assertEquals("Hello", res);
    }
}
