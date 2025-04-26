package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Order_equals_0_2_Test {

    @Test
    public void testEquals() {
        Order order1 = new Order();
        Order order2 = new Order();
        // Test case 1: Same object, but different values
        order1.equals(order2);
        // Test case 2: Different object, but same values
        order1 = new Order();
        order2 = new Order();
        order1.equals(order2);
        // Test case 3: Different object, but same values
        order1 = new Order();
        order2 = new Order();
        order1.equals(order2);
        // Test case 4: Null object
        try {
            order1.equals(null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            // expected
        }
        // Test case 5: Null object
        try {
            order1.equals(null);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            // expected
        }
        // Test case 6: Different object, but same values
        order1 = new Order();
        order2 = new Order();
        order1.equals(order2);
        // Test case 7: Different object, but same values
        order1 = new Order();
        order2 = new Order();
        order1.equals(order2);
        // Test case 8: Different object, but same values
        order1 = new Order();
        order2 = new Order();
        order1.equals(order2);
    }
}
