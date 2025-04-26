package com.ib.client;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_1_0_Test {

    @Test
    public void testInstrument() {
        // Setup
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        // Expected behavior
        String instrumentName = scannerSubscription.instrument();
        assertEquals("No row number specified", instrumentName, scannerSubscription.numberOfRows());
    }
}
