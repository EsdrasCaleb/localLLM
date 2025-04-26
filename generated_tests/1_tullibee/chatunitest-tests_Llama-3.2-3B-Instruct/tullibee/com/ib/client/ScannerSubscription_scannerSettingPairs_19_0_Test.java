package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_19_0_Test {

    @Test
    public void testScannerSettingPairs_Getter() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String scannerSettingPairsValue = scannerSubscription.scannerSettingPairs();
        // or any other expected value
        assertEquals("", scannerSettingPairsValue);
    }

    @Test
    public void testScannerSettingPairs_Getter_MultipleTimes() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String scannerSettingPairsValue1 = scannerSubscription.scannerSettingPairs();
        String scannerSettingPairsValue2 = scannerSubscription.scannerSettingPairs();
        assertEquals(scannerSettingPairsValue1, scannerSettingPairsValue2);
    }

    @Test
    public void testScannerSettingPairs_Setter() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scannerSettingPairs("new value");
        String scannerSettingPairsValue = scannerSubscription.scannerSettingPairs();
        assertEquals("new value", scannerSettingPairsValue);
    }

    @Test
    public void testScannerSettingPairs_Setter_MultipleTimes() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scannerSettingPairs("new value");
        scannerSubscription.scannerSettingPairs("another value");
        String scannerSettingPairsValue = scannerSubscription.scannerSettingPairs();
        assertEquals("another value", scannerSettingPairsValue);
    }
}
