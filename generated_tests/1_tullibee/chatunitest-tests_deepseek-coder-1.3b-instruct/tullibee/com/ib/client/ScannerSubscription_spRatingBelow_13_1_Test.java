package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_spRatingBelow_13_1_Test {

    @Test
    void testSpRatingBelow() {
        // Arrange
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String expected = "Expected value";
        // Act
        scannerSubscription.spRatingBelow(expected);
        String actual = scannerSubscription.spRatingBelow();
        // Assert
        assertEquals(expected, actual);
    }
}
