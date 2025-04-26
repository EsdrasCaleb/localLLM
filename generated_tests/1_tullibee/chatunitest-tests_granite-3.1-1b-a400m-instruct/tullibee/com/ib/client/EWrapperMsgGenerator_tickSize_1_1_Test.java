package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_tickSize_1_1_Test {

    @Test
    void testTickSize() {
        // Given
        String expectedOutput = "id=tickerId  Field=field  Size=size";
        // When
        String actualOutput = EWrapperMsgGenerator.tickSize(1, 1, 10);
        // Then
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testTickSizeWithDifferentFields() {
        // Given
        String expectedOutput = "id=tickerId  Field=10  Size=100";
        // When
        String actualOutput = EWrapperMsgGenerator.tickSize(1, 2, 10);
        // Then
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testTickSizeWithDifferentSizes() {
        // Given
        String expectedOutput = "id=tickerId  Field=100  Size=10";
        // When
        String actualOutput = EWrapperMsgGenerator.tickSize(1, 1, 100);
        // Then
        assertEquals(expectedOutput, actualOutput);
    }
}
