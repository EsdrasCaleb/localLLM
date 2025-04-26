package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_1_Test {

    @Test
    public void testAbovePrice() {
        // Arrange
        ScannerSubscription scannerSubscription = Mockito.spy(new ScannerSubscription());
        double expected = 100.0;
        double actual = 0.0;
        // Act
        Mockito.doReturn(expected).when(scannerSubscription).abovePrice();
        // Assert
        assertEquals(expected, scannerSubscription.abovePrice());
    }
}
