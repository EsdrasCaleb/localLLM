package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_40_1_Test {

    private String m_scannerSettingPairs;

    public void scannerSettingPairs(String settings) {
        m_scannerSettingPairs = settings;
    }

    @Test
    public void testScannerSettingPairs() {
        assertEquals("Setting1,Setting2", m_scannerSettingPairs);
    }
}
