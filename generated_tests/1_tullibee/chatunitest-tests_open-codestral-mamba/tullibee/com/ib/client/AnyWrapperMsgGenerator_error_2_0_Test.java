package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    public void testErrorMethod() {
        int id = 1;
        int errorCode = 404;
        String errorMsg = "Not Found";
        String expectedErrorMsg = "1 | 404 | Not Found";
        String actualErrorMsg = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expectedErrorMsg, actualErrorMsg);
    }
}
