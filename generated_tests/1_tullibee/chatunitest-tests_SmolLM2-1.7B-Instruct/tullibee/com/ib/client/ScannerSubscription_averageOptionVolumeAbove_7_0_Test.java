package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_7_0_Test {

    @Test
    public void testAverageOptionVolumeAbove() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        subscription.averageOptionVolumeAbove(5);
        // Act
        int result = subscription.averageOptionVolumeAbove();
        // Assert
        assertEquals(5, result);
    }
}
