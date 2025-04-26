package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_1_0_Test {

    @Test
    public void testErrorWithNonNullString() {
        String input = "Test Message";
        String result = AnyWrapperMsgGenerator.error(input);
        assertEquals(input, result);
    }

    @Test
    public void testErrorWithEmptyString() {
        String input = "";
        String result = AnyWrapperMsgGenerator.error(input);
        assertEquals(input, result);
    }

    @Test
    public void testErrorWithNullString() {
        String input = null;
        String result = AnyWrapperMsgGenerator.error(input);
        assertEquals(input, result);
    }
}
