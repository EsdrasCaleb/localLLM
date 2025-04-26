package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_2_Test {

    @Test
    public void testError() {
        String expected = "123 | 404 | Not Found";
        String result = AnyWrapperMsgGenerator.error(123, 404, "Not Found");
        assertEquals(expected, result);
    }
}
