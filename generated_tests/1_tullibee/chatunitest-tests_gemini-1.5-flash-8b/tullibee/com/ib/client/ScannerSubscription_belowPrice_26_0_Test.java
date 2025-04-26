package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_belowPrice_26_0_Test {

    @Test
    void testBelowPrice() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid price
        double price = 100.50;
        Method belowPriceMethod = ScannerSubscription.class.getDeclaredMethod("belowPrice", double.class);
        belowPriceMethod.setAccessible(true);
        belowPriceMethod.invoke(subscription, price);
        assertEquals(price, subscription.belowPrice());
        // Test with a zero price
        price = 0.0;
        belowPriceMethod.invoke(subscription, price);
        assertEquals(price, subscription.belowPrice());
        // Test with a negative price
        price = -10.5;
        belowPriceMethod.invoke(subscription, price);
        assertEquals(price, subscription.belowPrice());
        // Test with a price close to Double.MAX_VALUE
        price = Double.MAX_VALUE - 100;
        belowPriceMethod.invoke(subscription, price);
        assertEquals(price, subscription.belowPrice());
    }
}
