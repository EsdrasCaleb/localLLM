package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_7_0_Test {

    @Test
    public void averageOptionVolumeAboveTest() {
        // Arrange
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        int expectedAverageOptionVolume = 100;
        // Act
        int actualAverageOptionVolume = scannerSubscription.averageOptionVolumeAbove();
        // Assert
        assertEquals(expectedAverageOptionVolume, actualAverageOptionVolume);
    }
}
