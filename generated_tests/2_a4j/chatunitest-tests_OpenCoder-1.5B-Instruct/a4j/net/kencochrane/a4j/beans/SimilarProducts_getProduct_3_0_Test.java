package net.kencochrane.a4j.beans;

// Focal class
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class SimilarProducts_getProduct_3_0_Test {

    ArrayList simProducts;

    public String[] getProduct() {
        String[] retString = new String[simProducts.size()];
        if (simProducts.size() > 0)
            simProducts.toArray(retString);
        return retString;
    }

    public void setProduct(String[] newString) {
        simProducts = new ArrayList(newString.length);
        for (int i = 0; i < newString.length; i++) {
            simProducts.add(newString[i]);
        }
    }

    public ArrayList getProductsArray() {
        return simProducts;
    }
}

class SimilarProductsTest {

    private SimilarProducts similarProducts;

    @BeforeEach
    void setUp() {
        similarProducts = new SimilarProducts();
        ArrayList<String> products = new ArrayList<>();
        products.add("Product1");
        products.add("Product2");
        products.add("Product3");
        similarProducts.setProduct(products.toArray(new String[0]));
    }

    @Test
    void getProductTest() {
        String product = similarProducts.getProduct(1);
        assertEquals("Product2", product);
    }
}
