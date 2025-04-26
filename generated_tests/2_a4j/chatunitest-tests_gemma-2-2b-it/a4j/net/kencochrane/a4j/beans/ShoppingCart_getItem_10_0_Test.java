package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.ArgumentMatchers.any;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class ShoppingCart_getItem_10_0_Test {

    @Mock
    ShoppingCart shoppingCart;

    @InjectMocks
    ShoppingCart shoppingCartInjectMocks;

    @Test
    void testGetItem_Success() {
        String itemId = "123";
        Item item = new Item();
        item.setItemId("123");
        when(shoppingCart.getItem(itemId)).thenReturn(item);
        Item actualItem = shoppingCartInjectMocks.getItem(itemId);
        assertEquals(item, actualItem);
    }
}
