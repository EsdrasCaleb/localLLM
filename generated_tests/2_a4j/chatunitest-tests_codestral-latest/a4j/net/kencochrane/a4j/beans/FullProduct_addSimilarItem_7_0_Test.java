package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_addSimilarItem_7_0_Test {

    @InjectMocks
    private FullProduct fullProduct;

    @Mock
    private MiniProduct miniProduct;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        fullProduct.setSimilarItems(new ArrayList<>());
    }

    @Test
    public void testAddSimilarItem() {
        fullProduct.addSimilarItem(miniProduct);
        assertEquals(1, fullProduct.getSimilarItems().size());
        assertTrue(fullProduct.getSimilarItems().contains(miniProduct));
    }

    @Test
    public void testAddSimilarItem_NullItem() {
        assertThrows(NullPointerException.class, () -> fullProduct.addSimilarItem(null));
    }
}
