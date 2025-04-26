package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class FullProduct_addAccessory_6_0_Test {

    @InjectMocks
    private FullProduct fullProduct;

    @BeforeEach
    public void setup() {
        fullProduct = new FullProduct();
    }

    @Test
    public void testAddAccessory() {
        MiniProduct miniProduct = new MiniProduct();
        fullProduct.addAccessory(miniProduct);
        assertEquals(1, fullProduct.getAccessories().size());
    }

    @Test
    public void testAddAccessoryTwice() {
        MiniProduct miniProduct = new MiniProduct();
        fullProduct.addAccessory(miniProduct);
        fullProduct.addAccessory(miniProduct);
        assertEquals(2, fullProduct.getAccessories().size());
    }

    @Test
    public void testAddAccessoryNull() {
        MiniProduct miniProduct = null;
        assertThrows(NullPointerException.class, () -> fullProduct.addAccessory(miniProduct));
    }
}
