package net.kencochrane.a4j.DAO;

import static org.mockito.ArgumentMatchers.any;
import java.io.FileInputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.*;
import net.kencochrane.a4j.file.FileUtil;

public class Product_getProduct_0_0_Test {

    @InjectMocks
    private Product product;

    @Mock
    private FileUtil fileUtil;

    @Mock
    private JOXBeanInputStream joxIn;

    @Mock
    private Search search;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetProduct_FileInNull() throws Exception {
        when(fileUtil.fetchASINFile(any(), any(), any(), any())).thenReturn(null);
        FullProduct result = product.getProduct("asin1", "offer1", "page1");
        assertNull(result);
    }

    @Test
    void testGetProduct_ProductInfoBeanNull() throws Exception {
        FileInputStream fileIn = mock(FileInputStream.class);
        when(fileUtil.fetchASINFile(any(), any(), any(), any())).thenReturn(fileIn);
        when(new JOXBeanInputStream(fileIn).readObject(ProductInfo.class)).thenReturn(null);
        FullProduct result = product.getProduct("asin1", "offer1", "page1");
        assertNotNull(result);
        assertNull(result.getDetails());
    }

    @Test
    void testGetProduct_DetailsArrayEmpty() throws Exception {
        FileInputStream fileIn = mock(FileInputStream.class);
        when(fileUtil.fetchASINFile(any(), any(), any(), any())).thenReturn(fileIn);
        ProductInfo productInfoBean = mock(ProductInfo.class);
        when(new JOXBeanInputStream(fileIn).readObject(ProductInfo.class)).thenReturn(productInfoBean);
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[0]);
        FullProduct result = product.getProduct("asin1", "offer1", "page1");
        assertNotNull(result);
        assertNull(result.getDetails());
    }

    @Test
    void testGetProduct_AccessoriesNull() throws Exception {
        FileInputStream fileIn = mock(FileInputStream.class);
        when(fileUtil.fetchASINFile(any(), any(), any(), any())).thenReturn(fileIn);
        ProductInfo productInfoBean = mock(ProductInfo.class);
        ProductDetails productDetails = mock(ProductDetails.class);
        when(new JOXBeanInputStream(fileIn).readObject(ProductInfo.class)).thenReturn(productInfoBean);
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[] { productDetails });
        when(productDetails.getAccessories()).thenReturn(null);
        FullProduct result = product.getProduct("asin1", "offer1", "page1");
        assertNotNull(result);
        assertNotNull(result.getDetails());
        assertTrue(result.getAccessories().isEmpty());
    }

    @Test
    void testGetProduct_AccessoriesArrayEmpty() throws Exception {
        FileInputStream fileIn = mock(FileInputStream.class);
        when(fileUtil.fetchASINFile(any(), any(), any(), any())).thenReturn(fileIn);
        ProductInfo productInfoBean = mock(ProductInfo.class);
        ProductDetails productDetails = mock(ProductDetails.class);
        Accessories accessories = mock(Accessories.class);
        when(new JOXBeanInputStream(fileIn).readObject(ProductInfo.class)).thenReturn(productInfoBean);
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[] { productDetails });
        when(productDetails.getAccessories()).thenReturn(accessories);
        when(accessories.getAccessoryArray()).thenReturn(new ArrayList<>());
        FullProduct result = product.getProduct("asin1", "offer1", "page1");
        assertNotNull(result);
        assertNotNull(result.getDetails());
        assertTrue(result.getAccessories().isEmpty());
    }

    @Test
    void testGetProduct_AccessoriesFileInNull() throws Exception {
        FileInputStream fileIn = mock(FileInputStream.class);
        when(fileUtil.fetchASINFile(any(), any(), any(), any())).thenReturn(fileIn);
        ProductInfo productInfoBean = mock(ProductInfo.class);
        ProductDetails productDetails = mock(ProductDetails.class);
        Accessories accessories = mock(Accessories.class);
        ArrayList<String> accessoryArray = new ArrayList<>();
        accessoryArray.add("accessory1");
        when(new JOXBeanInputStream(fileIn).readObject(ProductInfo.class)).thenReturn(productInfoBean);
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[] { productDetails });
        when(productDetails.getAccessories()).thenReturn(accessories);
        when(accessories.getAccessoryArray()).thenReturn(accessoryArray);
        when(fileUtil.fetchAccessories(any(), any())).thenReturn(null);
        FullProduct result = product.getProduct("asin1", "offer1", "page1");
        assertNotNull(result);
        assertNotNull(result.getDetails());
        assertTrue(result.getAccessories().isEmpty());
    }
}
