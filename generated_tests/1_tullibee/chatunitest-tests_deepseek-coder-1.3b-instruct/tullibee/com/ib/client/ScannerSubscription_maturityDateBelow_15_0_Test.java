package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateBelow_15_0_Test {

    @Test
    void testMaturityDateBelow() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        String expected = "2022-01-01";
        // Act
        subscription.maturityDateBelow("2022-01-01");
        String result = subscription.maturityDateBelow();
        // Assert
        assertEquals(expected, result);
    }
}
