package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class ScannerSubscription_spRatingAbove_12_0_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @InjectMocks
    private ScannerSubscription scannerSubscriptionInstance;

    @Test
    void spRatingAbove() {
        // Arrange
        String expectedRating = "BBB";
        when(scannerSubscription.spRatingAbove()).thenReturn(expectedRating);
        // Act
        String actualRating = scannerSubscriptionInstance.spRatingAbove();
        // Assert
        assertEquals(expectedRating, actualRating);
    }
}
