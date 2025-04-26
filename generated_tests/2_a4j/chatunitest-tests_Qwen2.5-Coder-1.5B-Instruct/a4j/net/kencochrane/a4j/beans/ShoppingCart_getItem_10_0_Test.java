package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;

public class ShoppingCart_getItem_10_0_Test {

    private ShoppingCart shoppingCart;

    private Item mockItem;

    @BeforeEach
    public void setUp() {
        shoppingCart = new ShoppingCart();
        mockItem = mock(Item.class);
        // Corrected line: constructor Items in class net.kencochrane.a4j.beans.Items can be applied to given types;  required: no arguments  found: net.kencochrane.a4j.beans.Item  reason: actual and formal argument lists differ in length
        shoppingCart.setItems(new Items());
    }

    @Test
    public void getItem_returnsNullWhenNoMatchingItemFound() {
        when(mockItem.getItemId()).thenReturn("non-existing-id");
        assertNull(shoppingCart.getItem("non-existing-id"));
    }
}
