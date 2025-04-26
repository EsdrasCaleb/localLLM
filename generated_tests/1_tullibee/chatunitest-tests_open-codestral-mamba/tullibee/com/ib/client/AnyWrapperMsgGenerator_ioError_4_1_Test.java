package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_ioError_4_1_Test {

    @Test
    public void testIoError() {
        Exception ex = new Exception("Test Exception");
        try {
            Mockito.when(AnyWrapperMsgGenerator.error(ex)).thenReturn("Mocked Error Message");
            String result = AnyWrapperMsgGenerator.ioError(ex);
            assertEquals("Mocked Error Message", result);
        } catch (Exception e) {
            fail("Unexpected exception: " + e.getMessage());
        }
    }
}
