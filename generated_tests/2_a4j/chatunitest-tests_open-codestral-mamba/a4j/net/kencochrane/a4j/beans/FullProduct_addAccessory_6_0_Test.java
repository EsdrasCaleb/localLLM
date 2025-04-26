package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_addAccessory_6_0_Test {

    private FullProduct fullProduct;

    private MiniProduct miniProduct;

    @BeforeEach
    void setUp() {
        fullProduct = new FullProduct();
        miniProduct = Mockito.mock(MiniProduct.class);
    }

    @Test
    void addAccessory() {
        fullProduct.addAccessory(miniProduct);
        ArrayList<MiniProduct> accessories = fullProduct.getAccessories();
        assertEquals(1, accessories.size());
        assertEquals(miniProduct, accessories.get(0));
    }
}
