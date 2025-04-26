package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_addSimilarItem_7_1_Test {

    @Test
    void addSimilarItem() {
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        fullProduct.addSimilarItem(miniProduct);
        assertTrue(fullProduct.getSimilarItems().contains(miniProduct));
    }
}
