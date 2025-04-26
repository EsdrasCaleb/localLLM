package net.kencochrane.a4j.DAO;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.*;
import net.kencochrane.a4j.file.FileUtil;

@ExtendWith(MockitoExtension.class)
public class Product_getProduct_0_1_Test {

    @Mock
    private FileUtil fileUtil;

    @Mock
    private Search search;

    @InjectMocks
    private Product product;

    private String asin = "B00005N5PF";

    private String offer = "heavy";

    private String page = "1";

    @BeforeEach
    public void setUp() throws IOException {
        when(fileUtil.fetchASINFile(anyString(), anyString(), anyString(), anyString())).thenReturn(mock(FileInputStream.class));
        when(fileUtil.fetchAccessories(anyString(), any())).thenReturn(mock(FileInputStream.class));
    }

    @Test
    public void testGetProduct() throws Exception {
        ProductInfo productInfoBean = mock(ProductInfo.class);
        ProductDetails productDetails = mock(ProductDetails.class);
        Accessories accessories = mock(Accessories.class);
        ArrayList<ProductDetails> accessoryArray = new ArrayList<>();
        accessoryArray.add(productDetails);
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[] { productDetails });
        when(productDetails.getAccessories()).thenReturn(accessories);
        when(accessories.getAccessoryArray()).thenReturn(accessoryArray);
        when(productInfoBean.getProductsArrayList()).thenReturn(accessoryArray);
        when(search.SimilaritesSearch(anyString(), anyString())).thenReturn(productInfoBean);
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNotNull(fullProduct);
        assertEquals(productDetails, fullProduct.getDetails());
        assertEquals(1, fullProduct.getAccessories().size());
        assertEquals(1, fullProduct.getSimilarItems().size());
    }

    @Test
    public void testGetProductNullFileInput() throws Exception {
        when(fileUtil.fetchASINFile(anyString(), anyString(), anyString(), anyString())).thenReturn(null);
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNull(fullProduct);
    }

    @Test
    public void testGetProductNullProductInfoBean() throws Exception {
        when(fileUtil.fetchASINFile(anyString(), anyString(), anyString(), anyString())).thenReturn(mock(FileInputStream.class));
        when(fileUtil.fetchAccessories(anyString(), any())).thenReturn(mock(FileInputStream.class));
        when(search.SimilaritesSearch(anyString(), anyString())).thenReturn(null);
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNotNull(fullProduct);
        assertNull(fullProduct.getDetails());
        assertEquals(0, fullProduct.getAccessories().size());
        assertEquals(0, fullProduct.getSimilarItems().size());
    }

    @Test
    public void testGetProductEmptyAccessoryArray() throws Exception {
        ProductInfo productInfoBean = mock(ProductInfo.class);
        ProductDetails productDetails = mock(ProductDetails.class);
        Accessories accessories = mock(Accessories.class);
        ArrayList<ProductDetails> accessoryArray = new ArrayList<>();
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[] { productDetails });
        when(productDetails.getAccessories()).thenReturn(accessories);
        when(accessories.getAccessoryArray()).thenReturn(accessoryArray);
        when(search.SimilaritesSearch(anyString(), anyString())).thenReturn(productInfoBean);
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNotNull(fullProduct);
        assertEquals(productDetails, fullProduct.getDetails());
        assertEquals(0, fullProduct.getAccessories().size());
        assertEquals(0, fullProduct.getSimilarItems().size());
    }
}
