package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_15_0_Test {

    @Test
    public void testMaturityDateBelow_WithNoRows_ReturnsNull() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertNull(scannerSubscription.maturityDateBelow());
    }

    @Test
    public void testMaturityDateBelow_WithRows_ReturnsMaturityDate() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(1);
        scannerSubscription.maturityDateBelow("2022-01-01");
        assertEquals("2022-01-01", scannerSubscription.maturityDateBelow());
    }

    @Test
    public void testMaturityDateBelow_WithRowsAndAbovePrice_ReturnsMaturityDate() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(1);
        scannerSubscription.abovePrice(1000.0);
        scannerSubscription.maturityDateBelow();
        assertEquals("2022-01-01", scannerSubscription.maturityDateBelow());
    }

    @Test
    public void testMaturityDateBelow_WithRowsAndBelowPrice_ReturnsNull() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.numberOfRows(1);
        scannerSubscription.belowPrice(1000.0);
        assertNull(scannerSubscription.maturityDateBelow());
    }
}
