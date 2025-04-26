package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class FullProduct_addAccessory_6_0_Test {

    @Test
    public void testAddAccessory() throws NoSuchFieldException, IllegalAccessException {
        // Create a mock instance of MiniProduct
        MiniProduct miniProduct = Mockito.mock(MiniProduct.class);
        // Create an instance of FullProduct and set its accessories field to a new ArrayList
        FullProduct fullProduct = new FullProduct();
        fullProduct.setAccessories(new ArrayList<>());
        // Call the addAccessory method on the fullProduct object with the mock miniProduct as the argument
        fullProduct.addAccessory(miniProduct);
        // Verify that the accessory was added to the accessories list
        assertEquals(1, fullProduct.getAccessories().size());
        assertEquals(miniProduct, fullProduct.getAccessories().get(0));
    }
}
