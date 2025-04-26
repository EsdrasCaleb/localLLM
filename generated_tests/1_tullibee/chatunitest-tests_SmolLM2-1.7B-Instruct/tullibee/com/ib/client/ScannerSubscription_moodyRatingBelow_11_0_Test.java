package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_0_Test {

    @Test
    public void moodyRatingBelowTest() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("A");
        subscription.moodyRatingBelow("B");
        // Act
        String moodyRatingBelow = subscription.moodyRatingBelow();
        // Assert
        assertNotNull(moodyRatingBelow);
        assertEquals("B", moodyRatingBelow);
    }
}
