package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_3_Test {

    @Test
    public void testExcludeConvertible() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.excludeConvertible("yes");
        assertEquals("yes", subscription.excludeConvertible());
    }
}
