package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_7_1_Test {

    private ScannerSubscription scannerSubscription;

    @Test
    public void testAverageOptionVolumeAbove() {
        // Arrange
        scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Mockito.when(scannerSubscription.averageOptionVolumeAbove()).thenReturn(10);
        // Act
        int result = scannerSubscription.averageOptionVolumeAbove();
        // Assert
        assertEquals(10, result);
    }
}
