package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_40_0_Test {

    @Test
    void testScannerSettingPairs_NullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.scannerSettingPairs(null);
        assertNull(getScannerSettingPairs(subscription));
    }

    @Test
    void testScannerSettingPairs_EmptyInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.scannerSettingPairs("");
        assertEquals("", getScannerSettingPairs(subscription));
    }

    @Test
    void testScannerSettingPairs_ValidInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String input = "key1=value1;key2=value2";
        subscription.scannerSettingPairs(input);
        assertEquals(input, getScannerSettingPairs(subscription));
    }

    private String getScannerSettingPairs(ScannerSubscription subscription) {
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_scannerSettingPairs");
            field.setAccessible(true);
            return (String) field.get(subscription);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access m_scannerSettingPairs field: " + e.getMessage());
            // Should not reach here due to fail()
            return null;
        }
    }
}
