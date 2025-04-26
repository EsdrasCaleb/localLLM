package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_averageOptionVolumeAbove_28_2_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @Test
    public void testAverageOptionVolumeAbove() {
        // Arrange
        int expectedValue = 10;
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.averageOptionVolumeAbove(expectedValue);
        // Act
        int actualValue = scannerSubscription.averageOptionVolumeAbove();
        // Assert
        assertEquals(expectedValue, actualValue);
    }

    @Test
    public void testAverageOptionVolumeAbove_setsCorrectValue() {
        // Arrange
        int expectedValue = 10;
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.averageOptionVolumeAbove(expectedValue);
        // Act and Assert
        assertEquals(expectedValue, scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_setsOriginalValue() {
        // Arrange
        int originalValue = 20;
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.averageOptionVolumeAbove(originalValue);
        // Act and Assert
        assertEquals(originalValue, scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_setsNewValue() {
        // Arrange
        int originalValue = 20;
        int newValue = 10;
        scannerSubscription = new ScannerSubscription();
        scannerSubscription.averageOptionVolumeAbove(originalValue);
        // Act
        scannerSubscription.averageOptionVolumeAbove(newValue);
        // Assert
        assertEquals(newValue, scannerSubscription.averageOptionVolumeAbove());
    }
}
