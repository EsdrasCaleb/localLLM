package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class FullProduct_addSimilarItem_7_2_Test {

    @Mock
    FullProduct fullProduct;

    @InjectMocks
    FullProduct fullProductInjectMocks;

    @Test
    public void addSimilarItem() {
        MiniProduct miniProduct = new MiniProduct();
        when(fullProduct.getSimilarItems()).thenReturn(new ArrayList<>());
        fullProductInjectMocks.addSimilarItem(miniProduct);
        assertEquals(1, fullProductInjectMocks.getSimilarItems().size());
    }
}
