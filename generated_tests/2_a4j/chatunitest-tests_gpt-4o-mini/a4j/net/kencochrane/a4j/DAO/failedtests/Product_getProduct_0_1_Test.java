package net.kencochrane.a4j.DAO;

import java.io.FileInputStream;
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

public class Product_getProduct_0_1_Test {

    @InjectMocks
    private Product product;

    @Mock
    private FileUtil fileUtil;

    @Mock
    private JOXBeanInputStream joxIn;

    @Mock
    private ProductInfo productInfoBean;

    @Mock
    private Accessories accessories;

    @Mock
    private Search search;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetProduct_FileInNotNull_ProductDetailsAndAccessories() throws Exception {
        String asin = "testASIN";
        String offer = "testOffer";
        String page = "1";
        FileInputStream fileInputStream = mock(FileInputStream.class);
        when(fileUtil.fetchASINFile(asin, "heavy", offer, page)).thenReturn(fileInputStream);
        when(joxIn.readObject(ProductInfo.class)).thenReturn(productInfoBean);
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[] { new ProductDetails() });
        when(productInfoBean.getDetails()[0].getAccessories()).thenReturn(accessories);
        when(accessories.getAccessoryArray()).thenReturn(new ArrayList<>());
        when(search.SimilaritesSearch(asin, "1")).thenReturn(productInfoBean);
        when(productInfoBean.getProductsArrayList()).thenReturn(new ArrayList<>());
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNotNull(fullProduct);
        verify(fileUtil).fetchASINFile(asin, "heavy", offer, page);
        verify(joxIn).readObject(ProductInfo.class);
    }

    @Test
    public void testGetProduct_FileInNull_ReturnsNull() throws Exception {
        String asin = "testASIN";
        String offer = "testOffer";
        String page = "1";
        when(fileUtil.fetchASINFile(asin, "heavy", offer, page)).thenReturn(null);
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNull(fullProduct);
    }

    @Test
    public void testGetProduct_ProductInfoBeanNull_ReturnsFullProductWithNoDetails() throws Exception {
        String asin = "testASIN";
        String offer = "testOffer";
        String page = "1";
        FileInputStream fileInputStream = mock(FileInputStream.class);
        when(fileUtil.fetchASINFile(asin, "heavy", offer, page)).thenReturn(fileInputStream);
        when(joxIn.readObject(ProductInfo.class)).thenReturn(null);
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNotNull(fullProduct);
        assertNull(fullProduct.getDetails());
    }

    @Test
    public void testGetProduct_AccessoryArrayNotNullAndHasItems() throws Exception {
        String asin = "testASIN";
        String offer = "testOffer";
        String page = "1";
        FileInputStream fileInputStream = mock(FileInputStream.class);
        ArrayList<ProductDetails> accessoryList = new ArrayList<>();
        accessoryList.add(new ProductDetails());
        when(fileUtil.fetchASINFile(asin, "heavy", offer, page)).thenReturn(fileInputStream);
        when(joxIn.readObject(ProductInfo.class)).thenReturn(productInfoBean);
        when(productInfoBean.getDetails()).thenReturn(new ProductDetails[] { new ProductDetails() });
        when(productInfoBean.getDetails()[0].getAccessories()).thenReturn(accessories);
        when(accessories.getAccessoryArray()).thenReturn(accessoryList);
        when(fileUtil.fetchAccessories(asin, accessoryList)).thenReturn(fileInputStream);
        when(joxIn.readObject(ProductInfo.class)).thenReturn(productInfoBean);
        when(productInfoBean.getProductsArrayList()).thenReturn(new ArrayList<>());
        FullProduct fullProduct = product.getProduct(asin, offer, page);
        assertNotNull(fullProduct);
        verify(fileUtil).fetchAccessories(asin, accessoryList);
    }

    @Test
    public void testGetProduct_SimilarItemsFound() throws Exception {
        String asin = "testASIN";
        String offer = "testOffer";
        String page = "1";
        FileInputStream fileInputStream = mock(FileInputStream.class);
        ArrayList<ProductDetails> similarItemsList = new ArrayList<>();
        similarItemsList.add(new ProductDetails());
    }
}
