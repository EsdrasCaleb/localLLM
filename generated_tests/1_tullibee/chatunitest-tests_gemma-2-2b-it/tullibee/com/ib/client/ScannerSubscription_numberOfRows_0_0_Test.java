package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    void numberOfRows() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, scannerSubscription.numberOfRows());
        scannerSubscription.numberOfRows(1);
        assertEquals(1, scannerSubscription.numberOfRows());
    }
}
