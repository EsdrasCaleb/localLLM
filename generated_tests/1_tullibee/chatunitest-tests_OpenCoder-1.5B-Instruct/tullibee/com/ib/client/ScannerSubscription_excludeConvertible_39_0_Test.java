package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_39_0_Test {

    private ScannerSubscription scannerSubscription;

    @Test
    public void testExcludeConvertible() {
        scannerSubscription = Mockito.mock(ScannerSubscription.class);
        String c = "Y";
        scannerSubscription.excludeConvertible(c);
        Mockito.verify(scannerSubscription).excludeConvertible(c);
    }
}
