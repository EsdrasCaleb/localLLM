package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_addSimilarItem_7_0_Test {

    private FullProduct fullProduct;

    private MiniProduct miniProduct;

    @BeforeEach
    void setUp() {
        fullProduct = new FullProduct();
        miniProduct = mock(MiniProduct.class);
    }

    @Test
    void testAddSimilarItem() {
        fullProduct.addSimilarItem(miniProduct);
        assertTrue(fullProduct.getSimilarItems().contains(miniProduct));
    }
}
