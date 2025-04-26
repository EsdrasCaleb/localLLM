package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    void testError() {
        Exception ex = new Exception("Test Exception");
        String result = AnyWrapperMsgGenerator.error(ex);
        assertEquals("Error - Test Exception", result);
    }
}
