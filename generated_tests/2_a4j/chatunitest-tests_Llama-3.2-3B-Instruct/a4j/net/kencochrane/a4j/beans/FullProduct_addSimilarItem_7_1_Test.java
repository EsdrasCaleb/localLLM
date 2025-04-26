package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class FullProduct_addSimilarItem_7_1_Test {

    @ExtendWith(MockitoExtension.class)
    public static class FullProduct {

        private List<MiniProduct> similarItems = new ArrayList<>();

        public void addSimilarItem(MiniProduct miniProduct) {
            similarItems.add(miniProduct);
        }

        public List<MiniProduct> getSimilarItems() {
            return similarItems;
        }
    }

    private static class MiniProduct {
    }

    @Test
    public void testAddSimilarItem() {
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        fullProduct.addSimilarItem(miniProduct);
        assertEquals(1, fullProduct.getSimilarItems().size());
    }

    @Test
    public void testAddSimilarItemMultipleTimes() {
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        fullProduct.addSimilarItem(miniProduct);
        fullProduct.addSimilarItem(miniProduct);
        assertEquals(2, fullProduct.getSimilarItems().size());
    }

    @Test
    public void testAddSimilarItemNull() {
        FullProduct fullProduct = new FullProduct();
        assertThrows(NullPointerException.class, () -> fullProduct.addSimilarItem(null));
    }

    @Test
    public void testAddSimilarItemEmptyList() {
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        fullProduct.addSimilarItem(miniProduct);
        assertEquals(1, fullProduct.getSimilarItems().size());
    }
}
