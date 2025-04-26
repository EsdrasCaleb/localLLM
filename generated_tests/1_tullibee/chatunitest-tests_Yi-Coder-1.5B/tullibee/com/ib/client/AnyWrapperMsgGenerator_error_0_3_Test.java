package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_0_3_Test {

    @Test
    public void testError() {
        assertEquals("Error - java.lang.NullPointerException", AnyWrapperMsgGenerator.error(new NullPointerException()));
    }
}
