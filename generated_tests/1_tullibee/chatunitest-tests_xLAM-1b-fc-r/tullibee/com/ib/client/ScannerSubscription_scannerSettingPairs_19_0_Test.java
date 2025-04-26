package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_scannerSettingPairs_19_0_Test {

    @Test
    void scannerSettingPairsTest() {
        // Given
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedSettingPairs = "Expected setting pairs";
        // When
        subscription.scannerSettingPairs(expectedSettingPairs);
        // Then
        assertEquals(expectedSettingPairs, subscription.scannerSettingPairs());
    }
}
