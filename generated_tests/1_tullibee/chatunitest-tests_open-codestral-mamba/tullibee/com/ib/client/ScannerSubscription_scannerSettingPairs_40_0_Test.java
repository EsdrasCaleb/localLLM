package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_scannerSettingPairs_40_0_Test {

    @Test
    void testScannerSettingPairs() {
        ScannerSubscription subscription = new ScannerSubscription();
        String testVal = "testValue";
        subscription.scannerSettingPairs(testVal);
        assertEquals(testVal, subscription.scannerSettingPairs());
    }
}
