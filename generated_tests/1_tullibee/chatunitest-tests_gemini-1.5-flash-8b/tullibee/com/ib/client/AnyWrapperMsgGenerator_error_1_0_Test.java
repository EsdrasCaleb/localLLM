package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_1_0_Test {

    @Test
    void error_shouldReturnInputString() {
        String inputString = "This is an error message.";
        String expectedOutput = inputString;
        String actualOutput = AnyWrapperMsgGenerator.error(inputString);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void error_withEmptyString_shouldReturnEmptyString() {
        String inputString = "";
        String expectedOutput = inputString;
        String actualOutput = AnyWrapperMsgGenerator.error(inputString);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void error_withNullString_shouldReturnNull() {
        String inputString = null;
        String expectedOutput = null;
        String actualOutput = AnyWrapperMsgGenerator.error(inputString);
        assertEquals(expectedOutput, actualOutput);
    }
}
