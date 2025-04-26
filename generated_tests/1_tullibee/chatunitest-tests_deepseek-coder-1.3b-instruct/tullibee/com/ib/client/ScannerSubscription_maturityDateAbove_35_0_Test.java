package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_35_0_Test {

    @Test
    public void testMaturityDateAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        String testString = "2022-12-31";
        subscription.maturityDateAbove(testString);
        assertEquals(testString, subscription.maturityDateAbove());
    }
}
